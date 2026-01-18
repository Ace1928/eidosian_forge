import argparse
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def convert_udop_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    name_to_checkpoint_path = {'udop-large': '/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-224/pytorch_model.bin', 'udop-large-512': '/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512/pytorch_model.bin', 'udop-large-512-300k': '/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512-300k-steps/pytorch_model.bin'}
    checkpoint_path = name_to_checkpoint_path[model_name]
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print('Checkpoint path:', checkpoint_path)
    image_size = 512 if '512' in model_name else 224
    config = UdopConfig(decoder_start_token_id=0, image_size=image_size)
    model = UdopForConditionalGeneration(config)
    model.eval()
    state_dict = {k.replace('cell2dembedding', 'cell_2d_embedding'): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)
    assert missing_keys == ['encoder.embed_patches.proj.weight', 'encoder.embed_patches.proj.bias']
    assert unexpected_keys == ['pos_embed']
    tokenizer = UdopTokenizer.from_pretrained('t5-base', legacy=True)
    size = {'height': image_size, 'width': image_size}
    image_processor = LayoutLMv3ImageProcessor(image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD, size=size)
    processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)
    input_ids, bbox, image = prepare_dummy_inputs(tokenizer, image_processor)
    prompt = 'Question answering. In which year is the report made?'
    encoding = processor(images=get_image(), text=prompt, return_tensors='pt')
    input_ids = encoding.input_ids
    try:
        EXPECTED_INPUT_IDS = torch.tensor([[11860, 18243, 5, 86, 84, 215, 19, 8, 934, 263, 58, 1, 489, 27, 3838, 7363, 4083, 14536, 3430, 5686, 5911, 17161, 134, 2038, 27, 3838, 22, 7, 4688, 7, 10, 389, 18202, 21, 8, 11046, 37, 3733, 523, 11, 38, 2388, 1628, 3, 13133, 23334, 6, 8, 1656, 79, 3806, 21, 4040, 640, 27, 3838, 22, 7, 701, 16534, 6, 8, 3, 76, 2693, 18, 23015, 5644, 24, 380, 3, 6015, 6, 11, 8, 701, 24, 79, 482, 21, 3, 88, 684, 6, 43, 263, 27, 3838, 22, 7, 3635, 1157, 4089, 6, 2651, 12, 1547, 22, 7, 3265, 655, 5, 19, 27, 3838, 22, 7, 38, 2388, 257, 12, 36, 8, 465, 209, 13409, 12150, 1959, 16, 8, 684, 6, 6737, 57, 165, 126, 13409, 12150, 1623, 5, 71, 1100, 30298, 934, 65, 12566, 24, 27, 3838, 31, 7, 126, 13409, 12150, 1623, 33, 8, 10391, 1710, 859, 8, 420, 3733, 4968, 688, 2699, 16, 1547, 5, 27, 3838, 1217, 131, 99, 23, 179, 6064, 24, 6, 590, 28, 3, 11600, 1456, 701, 6, 175, 9443, 2557, 3635, 92, 1262, 8, 3409, 13, 2186, 3, 27908, 1784, 190, 8, 3, 5771, 17, 13281, 4005, 13, 5086, 11, 13066, 1170, 5, 10826, 16309, 134, 3, 2, 276, 26, 3, 55, 391, 13570, 5, 10315, 309, 3577, 19114, 371, 4254, 5121, 5055, 6245, 3, 10047, 3162, 58, 3, 9, 61, 1713, 2703, 476, 667, 25158, 301, 6058, 6038, 476, 3765, 9149, 10, 4893, 1303, 1986, 5, 13580, 7, 8224, 28244, 7, 5, 76, 75, 7, 89, 5, 15, 1259, 87, 7171, 7, 87, 7, 29, 115, 226, 4305, 2773, 1]])
        torch.testing.assert_close(EXPECTED_INPUT_IDS, input_ids)
        bbox = encoding.bbox.float()
        pixel_values = encoding.pixel_values
    except Exception:
        print("Input_ids don't match, preparing dummy inputs")
        input_ids, bbox, pixel_values = prepare_dummy_inputs(tokenizer, image_processor)
    print('Testing single forward pass..')
    with torch.no_grad():
        decoder_input_ids = torch.tensor([[101]])
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        print('Shape of logits:', outputs.logits.shape)
        print('First values of logits:', outputs.logits[0, :3, :3])
    try:
        assert torch.allclose(outputs.logits[0, :3, :3], torch.tensor([[-18.5262, 1.5087, -15.7051]]), atol=0.0001)
        print('Looks ok!')
    except Exception:
        print("logits don't match let's try to generate")
    print('Testing generation...')
    model_kwargs = {'bbox': bbox, 'pixel_values': pixel_values}
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)
    print('Generated:', tokenizer.batch_decode(outputs, skip_special_tokens=True))
    print('Testing generation with original inputs...')
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename='input_ids_udop.pt', repo_type='dataset')
    input_ids = torch.load(filepath)
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename='bbox_udop.pt', repo_type='dataset')
    bbox = torch.load(filepath)
    pixel_values_filename = 'pixel_values_udop_512.pt' if '512' in model_name else 'pixel_values_udop_224.pt'
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename=pixel_values_filename, repo_type='dataset')
    pixel_values = torch.load(filepath)
    print('Decoded input ids:', tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print('Bbox shape:', bbox.shape)
    model_kwargs = {'bbox': bbox, 'pixel_values': pixel_values}
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print('Generated:', generated_text)
    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        tokenizer.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub(f'microsoft/{model_name}')
        processor.push_to_hub(f'microsoft/{model_name}')