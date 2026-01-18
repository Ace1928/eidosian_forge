import collections
import urllib.parse
import urllib.request
def can_fetch(self, useragent, url):
    """using the parsed robots.txt decide if useragent can fetch url"""
    if self.disallow_all:
        return False
    if self.allow_all:
        return True
    if not self.last_checked:
        return False
    parsed_url = urllib.parse.urlparse(urllib.parse.unquote(url))
    url = urllib.parse.urlunparse(('', '', parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))
    url = urllib.parse.quote(url)
    if not url:
        url = '/'
    for entry in self.entries:
        if entry.applies_to(useragent):
            return entry.allowance(url)
    if self.default_entry:
        return self.default_entry.allowance(url)
    return True