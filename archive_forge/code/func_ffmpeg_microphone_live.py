import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union
import numpy as np
def ffmpeg_microphone_live(sampling_rate: int, chunk_length_s: float, stream_chunk_s: Optional[int]=None, stride_length_s: Optional[Union[Tuple[float, float], float]]=None, format_for_conversion: str='f32le'):
    """
    Helper function to read audio from the microphone file through ffmpeg. This will output `partial` overlapping
    chunks starting from `stream_chunk_s` (if it is defined) until `chunk_length_s` is reached. It will make use of
    striding to avoid errors on the "sides" of the various chunks.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned. This includes the eventual striding.
        stream_chunk_s (`float` or `int`)
            The length of the minimal temporary audio to be returned.
        stride_length_s (`float` or `int` or `(float, float)`, *optional*, defaults to `None`)
            The length of the striding to be used. Stride is used to provide context to a model on the (left, right) of
            an audio sample but without using that part to actually make the prediction. Setting this does not change
            the length of the chunk.
        format_for_conversion (`str`, defalts to `f32le`)
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
    Return:
        A generator yielding dictionaries of the following form

        `{"sampling_rate": int, "raw": np.array(), "partial" bool}` With optionnally a `"stride" (int, int)` key if
        `stride_length_s` is defined.

        `stride` and `raw` are all expressed in `samples`, and `partial` is a boolean saying if the current yield item
        is a whole chunk, or a partial temporary result to be later replaced by another larger chunk.


    """
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s
    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)
    if format_for_conversion == 's16le':
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == 'f32le':
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f'Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`')
    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]
    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample
    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        item['raw'] = np.frombuffer(item['raw'], dtype=dtype)
        item['stride'] = (item['stride'][0] // size_of_sample, item['stride'][1] // size_of_sample)
        item['sampling_rate'] = sampling_rate
        audio_time += delta
        if datetime.datetime.now() > audio_time + 10 * delta:
            continue
        yield item