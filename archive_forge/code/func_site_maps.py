import collections
import urllib.parse
import urllib.request
def site_maps(self):
    if not self.sitemaps:
        return None
    return self.sitemaps