import inspect
import logging
import urllib.parse
def safe_urlsplit(url):
    """This is a hack to prevent the regular urlsplit from splitting around question marks.

    A question mark (?) in a URL typically indicates the start of a
    querystring, and the standard library's urlparse function handles the
    querystring separately.  Unfortunately, question marks can also appear
    _inside_ the actual URL for some schemas like S3, GS.

    Replaces question marks with a special placeholder substring prior to
    splitting.  This work-around behavior is disabled in the unlikely event the
    placeholder is already part of the URL.  If this affects you, consider
    changing the value of QUESTION_MARK_PLACEHOLDER to something more suitable.

    See Also
    --------
    https://bugs.python.org/issue43882
    https://github.com/python/cpython/blob/3.7/Lib/urllib/parse.py
    https://github.com/RaRe-Technologies/smart_open/issues/285
    https://github.com/RaRe-Technologies/smart_open/issues/458
    smart_open/utils.py:QUESTION_MARK_PLACEHOLDER
    """
    sr = urllib.parse.urlsplit(url, allow_fragments=False)
    placeholder = None
    if sr.scheme in WORKAROUND_SCHEMES and '?' in url and (QUESTION_MARK_PLACEHOLDER not in url):
        placeholder = QUESTION_MARK_PLACEHOLDER
        url = url.replace('?', placeholder)
        sr = urllib.parse.urlsplit(url, allow_fragments=False)
    if placeholder is None:
        return sr
    path = sr.path.replace(placeholder, '?')
    return urllib.parse.SplitResult(sr.scheme, sr.netloc, path, '', '')