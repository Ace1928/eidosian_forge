import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth
@router.post('/midi-to-wav', tags=['MIDI'])
def midi_to_wav(body: MidiToWavBody):
    """
    Install fluidsynth first, see more: https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
    """
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    if not body.wav_path.startswith('midi/'):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'bad output path')
    fs = FluidSynth(body.sound_font_path)
    fs.midi_to_audio(body.midi_path, body.wav_path)
    return 'success'