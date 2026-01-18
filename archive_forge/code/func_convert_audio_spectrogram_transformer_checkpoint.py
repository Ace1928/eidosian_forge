import argparse
import json
from pathlib import Path
import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.utils import logging
@torch.no_grad()
def convert_audio_spectrogram_transformer_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Audio Spectrogram Transformer structure.
    """
    config = get_audio_spectrogram_transformer_config(model_name)
    model_name_to_url = {'ast-finetuned-audioset-10-10-0.4593': 'https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1', 'ast-finetuned-audioset-10-10-0.450': 'https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1', 'ast-finetuned-audioset-10-10-0.448': 'https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1', 'ast-finetuned-audioset-10-10-0.448-v2': 'https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1', 'ast-finetuned-audioset-12-12-0.447': 'https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1', 'ast-finetuned-audioset-14-14-0.443': 'https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1', 'ast-finetuned-audioset-16-16-0.442': 'https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1', 'ast-finetuned-speech-commands-v2': 'https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1'}
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')
    remove_keys(state_dict)
    new_state_dict = convert_state_dict(state_dict, config)
    model = ASTForAudioClassification(config)
    model.eval()
    model.load_state_dict(new_state_dict)
    mean = -4.2677393 if 'speech-commands' not in model_name else -6.845978
    std = 4.5689974 if 'speech-commands' not in model_name else 5.5654526
    max_length = 1024 if 'speech-commands' not in model_name else 128
    feature_extractor = ASTFeatureExtractor(mean=mean, std=std, max_length=max_length)
    if 'speech-commands' in model_name:
        dataset = load_dataset('speech_commands', 'v0.02', split='validation')
        waveform = dataset[0]['audio']['array']
    else:
        filepath = hf_hub_download(repo_id='nielsr/audio-spectogram-transformer-checkpoint', filename='sample_audio.flac', repo_type='dataset')
        waveform, _ = torchaudio.load(filepath)
        waveform = waveform.squeeze().numpy()
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    if model_name == 'ast-finetuned-audioset-10-10-0.4593':
        expected_slice = torch.tensor([-0.876, -7.0042, -8.6602])
    elif model_name == 'ast-finetuned-audioset-10-10-0.450':
        expected_slice = torch.tensor([-1.1986, -7.0903, -8.2718])
    elif model_name == 'ast-finetuned-audioset-10-10-0.448':
        expected_slice = torch.tensor([-2.6128, -8.008, -9.4344])
    elif model_name == 'ast-finetuned-audioset-10-10-0.448-v2':
        expected_slice = torch.tensor([-1.508, -7.4534, -8.8917])
    elif model_name == 'ast-finetuned-audioset-12-12-0.447':
        expected_slice = torch.tensor([-0.505, -6.5833, -8.0843])
    elif model_name == 'ast-finetuned-audioset-14-14-0.443':
        expected_slice = torch.tensor([-0.3826, -7.0336, -8.2413])
    elif model_name == 'ast-finetuned-audioset-16-16-0.442':
        expected_slice = torch.tensor([-1.2113, -6.9101, -8.347])
    elif model_name == 'ast-finetuned-speech-commands-v2':
        expected_slice = torch.tensor([6.1589, -8.0566, -8.7984])
    else:
        raise ValueError('Unknown model name')
    if not torch.allclose(logits[0, :3], expected_slice, atol=0.0001):
        raise ValueError("Logits don't match")
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving feature extractor to {pytorch_dump_folder_path}')
        feature_extractor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing model and feature extractor to the hub...')
        model.push_to_hub(f'MIT/{model_name}')
        feature_extractor.push_to_hub(f'MIT/{model_name}')