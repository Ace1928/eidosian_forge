import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def initsysfonts_unix(path='fc-list'):
    """use the fc-list from fontconfig to get a list of fonts"""
    fonts = {}
    if sys.platform == 'emscripten':
        return fonts
    try:
        proc = subprocess.run([path, ':', 'file', 'family', 'style'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=1)
    except FileNotFoundError:
        warnings.warn(f"'{path}' is missing, system fonts cannot be loaded on your platform")
    except subprocess.TimeoutExpired:
        warnings.warn(f"Process running '{path}' timed-out! System fonts cannot be loaded on your platform")
    except subprocess.CalledProcessError as e:
        warnings.warn(f"'{path}' failed with error code {e.returncode}! System fonts cannot be loaded on your platform. Error log is:\n{e.stderr}")
    else:
        for entry in proc.stdout.decode('ascii', 'ignore').splitlines():
            try:
                _parse_font_entry_unix(entry, fonts)
            except ValueError:
                pass
    return fonts