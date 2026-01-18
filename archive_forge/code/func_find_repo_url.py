import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen
from breezy.errors import BzrError
from breezy.trace import note
from breezy.urlutils import InvalidURL
def find_repo_url(data):
    for key, value in data['info']['project_urls'].items():
        if key == 'Repository':
            note('Found repository URL %s for pypi project %s', value, name)
            return value
        parsed_url = urlparse(value)
        if parsed_url.hostname == 'github.com' and parsed_url.path.strip('/').count('/') == 1:
            return value