import urllib.parse
import urllib.request
from urllib.error import URLError, HTTPError
import os
import re
import tempfile
from contextlib import contextmanager
@contextmanager
def file_or_url_context(resource_name):
    """Yield name of file from the given resource (i.e. file or url)."""
    if is_url(resource_name):
        url_components = urllib.parse.urlparse(resource_name)
        _, ext = os.path.splitext(url_components.path)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                with urllib.request.urlopen(resource_name) as u:
                    f.write(u.read())
            yield f.name
        except (URLError, HTTPError):
            os.remove(f.name)
            raise
        except (FileNotFoundError, FileExistsError, PermissionError, BaseException):
            raise
        else:
            os.remove(f.name)
    else:
        yield resource_name