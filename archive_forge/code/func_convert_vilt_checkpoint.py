import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_vilt_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViLT structure.
    """
    config = ViltConfig(image_size=384, patch_size=32, tie_word_embeddings=False)
    mlm_model = False
    vqa_model = False
    nlvr_model = False
    irtr_model = False
    if 'vqa' in checkpoint_url:
        vqa_model = True
        config.num_labels = 3129
        repo_id = 'huggingface/label-files'
        filename = 'vqa2-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        model = ViltForQuestionAnswering(config)
    elif 'nlvr' in checkpoint_url:
        nlvr_model = True
        config.num_labels = 2
        config.id2label = {0: 'False', 1: 'True'}
        config.label2id = {v: k for k, v in config.id2label.items()}
        config.modality_type_vocab_size = 3
        model = ViltForImagesAndTextClassification(config)
    elif 'irtr' in checkpoint_url:
        irtr_model = True
        model = ViltForImageAndTextRetrieval(config)
    elif 'mlm_itm' in checkpoint_url:
        mlm_model = True
        model = ViltForMaskedLM(config)
    else:
        raise ValueError('Unknown model type')
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['state_dict']
    rename_keys = create_rename_keys(config, vqa_model, nlvr_model, irtr_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)
    if mlm_model or irtr_model:
        ignore_keys = ['itm_score.fc.weight', 'itm_score.fc.bias']
        for k in ignore_keys:
            state_dict.pop(k, None)
    model.eval()
    if mlm_model:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert missing_keys == ['mlm_score.decoder.bias']
    else:
        model.load_state_dict(state_dict)
    image_processor = ViltImageProcessor(size=384)
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    processor = ViltProcessor(image_processor, tokenizer)
    if nlvr_model:
        image1 = Image.open(requests.get('https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg', stream=True).raw)
        image2 = Image.open(requests.get('https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg', stream=True).raw)
        text = 'The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.'
        encoding_1 = processor(image1, text, return_tensors='pt')
        encoding_2 = processor(image2, text, return_tensors='pt')
        outputs = model(input_ids=encoding_1.input_ids, pixel_values=encoding_1.pixel_values, pixel_values_2=encoding_2.pixel_values)
    else:
        image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw)
        if mlm_model:
            text = 'a bunch of [MASK] laying on a [MASK].'
        else:
            text = 'How many cats are there?'
        encoding = processor(image, text, return_tensors='pt')
        outputs = model(**encoding)
    if mlm_model:
        expected_shape = torch.Size([1, 11, 30522])
        expected_slice = torch.tensor([-12.5061, -12.5123, -12.5174])
        assert outputs.logits.shape == expected_shape
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=0.0001)
        predicted_id = outputs.logits[0, 4, :].argmax(-1).item()
        assert tokenizer.decode([predicted_id]) == 'cats'
    elif vqa_model:
        expected_shape = torch.Size([1, 3129])
        expected_slice = torch.tensor([-15.9495, -18.1472, -10.3041])
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001)
        assert outputs.logits.shape == expected_shape
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=0.0001)
        predicted_idx = outputs.logits.argmax(-1).item()
        assert model.config.id2label[predicted_idx] == '2'
    elif nlvr_model:
        expected_shape = torch.Size([1, 2])
        expected_slice = torch.tensor([-2.8721, 2.1291])
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001)
        assert outputs.logits.shape == expected_shape
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model and processor to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)