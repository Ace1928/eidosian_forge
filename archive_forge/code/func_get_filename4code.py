import codecs
import hashlib
import io
import json
import os
import sys
import atexit
import shutil
import tempfile
def get_filename4code(module, content, ext=None):
    """Generate filename based on content

    The function ensures that the (temporary) directory exists, so that the
    file can be written.

    By default, the directory won't be cleaned up,
    so a filter can use the directory as a cache and
    decide not to regenerate if there's no change.

    In case the user preferres the files to be temporary files,
    an environment variable `PANDOCFILTER_CLEANUP` can be set to
    any non-empty value such as `1` to
    make sure the directory is created in a temporary location and removed
    after finishing the filter. In this case there's no caching and files
    will be regenerated each time the filter is run.

    Example:
        filename = get_filename4code("myfilter", code)
    """
    if os.getenv('PANDOCFILTER_CLEANUP'):
        imagedir = tempfile.mkdtemp(prefix=module)
        atexit.register(lambda: shutil.rmtree(imagedir))
    else:
        imagedir = module + '-images'
    fn = hashlib.sha1(content.encode(sys.getfilesystemencoding())).hexdigest()
    try:
        os.makedirs(imagedir, exist_ok=True)
        sys.stderr.write('Created directory ' + imagedir + '\n')
    except OSError:
        sys.stderr.write('Could not create directory "' + imagedir + '"\n')
    if ext:
        fn += '.' + ext
    return os.path.join(imagedir, fn)