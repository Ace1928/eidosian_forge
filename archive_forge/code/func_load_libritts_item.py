import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
def load_libritts_item(fileid: str, path: str, ext_audio: str, ext_original_txt: str, ext_normalized_txt: str) -> Tuple[Tensor, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split('_')
    utterance_id = fileid
    normalized_text = utterance_id + ext_normalized_txt
    normalized_text = os.path.join(path, speaker_id, chapter_id, normalized_text)
    original_text = utterance_id + ext_original_txt
    original_text = os.path.join(path, speaker_id, chapter_id, original_text)
    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
    waveform, sample_rate = torchaudio.load(file_audio)
    with open(original_text) as ft:
        original_text = ft.readline()
    with open(normalized_text, 'r') as ft:
        normalized_text = ft.readline()
    return (waveform, sample_rate, original_text, normalized_text, int(speaker_id), int(chapter_id), utterance_id)