from ctypes import memmove, byref, c_uint32, sizeof, cast, c_void_p, create_string_buffer, POINTER, c_char, \
from pyglet.libs.darwin import cf, CFSTR
from pyglet.libs.darwin.coreaudio import kCFURLPOSIXPathStyle, AudioStreamBasicDescription, ca, ExtAudioFileRef, \
from pyglet.media import StreamingSource, StaticSource
from pyglet.media.codecs import AudioFormat, MediaDecoder, AudioData
@staticmethod
def convert_format(original_desc, bitdepth=16):
    adesc = AudioStreamBasicDescription()
    adesc.mSampleRate = original_desc.mSampleRate
    adesc.mFormatID = kAudioFormatLinearPCM
    adesc.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
    adesc.mChannelsPerFrame = original_desc.mChannelsPerFrame
    adesc.mBitsPerChannel = bitdepth
    adesc.mBytesPerPacket = original_desc.mChannelsPerFrame * adesc.mBitsPerChannel // 8
    adesc.mFramesPerPacket = 1
    adesc.mBytesPerFrame = adesc.mBytesPerPacket
    return adesc