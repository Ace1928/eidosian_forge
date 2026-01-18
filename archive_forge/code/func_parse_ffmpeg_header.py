import re
import threading
import time
from ._utils import logger
def parse_ffmpeg_header(text):
    lines = text.splitlines()
    meta = {}
    ver = lines[0].split('version', 1)[-1].split('Copyright')[0]
    meta['ffmpeg_version'] = ver.strip() + ' ' + lines[1].strip()
    videolines = [l for l in lines if l.lstrip().startswith('Stream ') and ' Video: ' in l]
    line = videolines[0]
    meta['codec'] = line.split('Video: ', 1)[-1].lstrip().split(' ', 1)[0].strip()
    meta['pix_fmt'] = re.split(',\\s*(?![^()]*\\))', line.split('Video: ', 1)[-1])[1].strip()
    audiolines = [l for l in lines if l.lstrip().startswith('Stream ') and ' Audio: ' in l]
    if len(audiolines) > 0:
        audio_line = audiolines[0]
        meta['audio_codec'] = audio_line.split('Audio: ', 1)[-1].lstrip().split(' ', 1)[0].strip()
    fps = 0
    for line in [videolines[0]]:
        matches = re.findall(' ([0-9]+\\.?[0-9]*) (fps)', line)
        if matches:
            fps = float(matches[0][0].strip())
    meta['fps'] = fps
    line = videolines[0]
    match = re.search(' [0-9]*x[0-9]*(,| )', line)
    parts = line[match.start():match.end() - 1].split('x')
    meta['source_size'] = tuple(map(int, parts))
    line = videolines[-1]
    match = re.search(' [0-9]*x[0-9]*(,| )', line)
    parts = line[match.start():match.end() - 1].split('x')
    meta['size'] = tuple(map(int, parts))
    if meta['source_size'] != meta['size']:
        logger.warning('The frame size for reading {} is different from the source frame size {}.'.format(meta['size'], meta['source_size']))
    reo_rotate = re.compile('rotate\\s+:\\s([0-9]+)')
    match = reo_rotate.search(text)
    rotate = 0
    if match is not None:
        rotate = match.groups()[0]
    meta['rotate'] = int(rotate)
    line = [l for l in lines if 'Duration: ' in l][0]
    match = re.search(' [0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9]', line)
    duration = 0
    if match is not None:
        hms = line[match.start() + 1:match.end()].split(':')
        duration = cvsecs(*hms)
    meta['duration'] = duration
    return meta