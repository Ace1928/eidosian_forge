from .build import build
import os
def join_speech(utt1, utt2):
    """
    Joins two conversations using a '
'
    Join texts and doesn't care about segments
    Assumption: utt1 is the one being popped off from the stack and the utt2
                is the one trying to be added. This is so that I check for 'ctr'
                field only in one of the parameters.
    :param utt1:
        An utterance (json object) from either USER or ASSISTANT
    :param utt2:
        Next utterance(json object) from a speaker same as utt1
    """
    new_utt = {}
    new_utt['index'] = utt1['index']
    new_utt['text'] = utt1['text'] + '\n' + utt2['text']
    new_utt['speaker'] = utt1['speaker']
    if 'ctr' in utt1:
        new_utt['ctr'] = utt1['ctr'] + 1
    else:
        new_utt['ctr'] = 2
    return new_utt