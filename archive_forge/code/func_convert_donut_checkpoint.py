import argparse
import torch
from datasets import load_dataset
from donut import DonutModel
from transformers import (
def convert_donut_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    original_model = DonutModel.from_pretrained(model_name).eval()
    encoder_config, decoder_config = get_configs(original_model)
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()
    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)
    dataset = load_dataset('hf-internal-testing/example-documents')
    image = dataset['test'][0]['image'].convert('RGB')
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name, from_slow=True)
    image_processor = DonutImageProcessor(do_align_long_axis=original_model.config.align_long_axis, size=original_model.config.input_size[::-1])
    processor = DonutProcessor(image_processor, tokenizer)
    pixel_values = processor(image, return_tensors='pt').pixel_values
    if model_name == 'naver-clova-ix/donut-base-finetuned-docvqa':
        task_prompt = '<s_docvqa><s_question>{user_input}</s_question><s_answer>'
        question = 'When is the coffee break?'
        task_prompt = task_prompt.replace('{user_input}', question)
    elif model_name == 'naver-clova-ix/donut-base-finetuned-rvlcdip':
        task_prompt = '<s_rvlcdip>'
    elif model_name in ['naver-clova-ix/donut-base-finetuned-cord-v1', 'naver-clova-ix/donut-base-finetuned-cord-v1-2560']:
        task_prompt = '<s_cord>'
    elif model_name == 'naver-clova-ix/donut-base-finetuned-cord-v2':
        task_prompt = 's_cord-v2>'
    elif model_name == 'naver-clova-ix/donut-base-finetuned-zhtrainticket':
        task_prompt = '<s_zhtrainticket>'
    elif model_name in ['naver-clova-ix/donut-proto', 'naver-clova-ix/donut-base']:
        task_prompt = 'hello world'
    else:
        raise ValueError('Model name not supported')
    prompt_tensors = original_model.decoder.tokenizer(task_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    assert torch.allclose(original_patch_embed, patch_embeddings, atol=0.001)
    original_last_hidden_state = original_model.encoder(pixel_values)
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=0.01)
    original_logits = original_model(pixel_values, prompt_tensors, None).logits
    logits = model(pixel_values, decoder_input_ids=prompt_tensors).logits
    assert torch.allclose(original_logits, logits, atol=0.001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub('nielsr/' + model_name.split('/')[-1], commit_message='Update model')
        processor.push_to_hub('nielsr/' + model_name.split('/')[-1], commit_message='Update model')