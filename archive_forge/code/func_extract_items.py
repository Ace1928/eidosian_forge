import mf2py
def extract_items(self, html, base_url=None):
    for obj in mf2py.parse(html, html_parser='lxml', url=base_url)['items']:
        yield obj