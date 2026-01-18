import argparse
import torch
from huggingface_hub import hf_hub_download
from nougat import NougatModel
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.checkpoint import get_checkpoint
from PIL import Image
from transformers import (
def convert_nougat_checkpoint(model_tag, pytorch_dump_folder_path=None, push_to_hub=False):
    checkpoint_path = get_checkpoint(None, model_tag)
    original_model = NougatModel.from_pretrained(checkpoint_path)
    original_model.eval()
    encoder_config, decoder_config = get_configs(original_model)
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()
    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)
    filepath = hf_hub_download(repo_id='ysharma/nougat', filename='input/nougat.pdf', repo_type='space')
    images = rasterize_paper(pdf=filepath, return_pil=True)
    image = Image.open(images[0])
    tokenizer_file = checkpoint_path / 'tokenizer.json'
    tokenizer = NougatTokenizerFast(tokenizer_file=str(tokenizer_file))
    tokenizer.pad_token = '<pad>'
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.model_max_length = original_model.config.max_length
    size = {'height': original_model.config.input_size[0], 'width': original_model.config.input_size[1]}
    image_processor = NougatImageProcessor(do_align_long_axis=original_model.config.align_long_axis, size=size)
    processor = NougatProcessor(image_processor=image_processor, tokenizer=tokenizer)
    pixel_values = processor(image, return_tensors='pt').pixel_values
    original_pixel_values = original_model.encoder.prepare_input(image).unsqueeze(0)
    assert torch.allclose(original_pixel_values, pixel_values)
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    assert torch.allclose(original_patch_embed, patch_embeddings)
    original_last_hidden_state = original_model.encoder(pixel_values)
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=0.01)
    original_embeddings = original_model.decoder.model.model.decoder.embed_tokens
    embeddings = model.decoder.model.decoder.embed_tokens
    assert torch.allclose(original_embeddings.weight, embeddings.weight, atol=0.001)
    prompt = 'hello world'
    decoder_input_ids = original_model.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    original_logits = original_model(image_tensors=pixel_values, decoder_input_ids=decoder_input_ids, attention_mask=decoder_attention_mask).logits
    logits = model(pixel_values, decoder_input_ids=decoder_input_ids[:, :-1], decoder_attention_mask=decoder_attention_mask[:, :-1]).logits
    assert torch.allclose(original_logits, logits, atol=0.001)
    outputs = model.generate(pixel_values, min_length=1, max_length=30, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True, bad_words_ids=[[tokenizer.unk_token_id]], return_dict_in_generate=True, do_sample=False)
    generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    if model_tag == '0.1.0-base':
        expected_generation = '# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lblec'
    elif model_tag == '0.1.0-small':
        expected_generation = '# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lble'
    else:
        raise ValueError(f'Unexpected model tag: {model_tag}')
    assert generated == expected_generation
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        tag_to_name = {'0.1.0-base': 'nougat-base', '0.1.0-small': 'nougat-small'}
        model_name = tag_to_name[model_tag]
        model.push_to_hub(f'facebook/{model_name}')
        processor.push_to_hub(f'facebook/{model_name}')