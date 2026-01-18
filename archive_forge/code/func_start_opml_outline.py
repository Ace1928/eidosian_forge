from __future__ import annotations
import copy
from . import common, dates
def start_opml_outline(self, attrs: dict[str, str]) -> None:
    if attrs.get('text', '').strip():
        title = attrs['text'].strip()
    else:
        title = attrs.get('title', '').strip()
    url = None
    append_to = None
    if 'xmlurl' in attrs:
        url = attrs.get('xmlurl', '').strip()
        append_to = 'feeds'
        if attrs.get('type', '').strip().lower() == 'source':
            append_to = 'lists'
    elif attrs.get('type', '').lower() in ('link', 'include'):
        append_to = 'lists'
        url = attrs.get('url', '').strip()
    elif title:
        self.hierarchy.append(title)
        return
    if not url and 'htmlurl' in attrs:
        url = attrs['htmlurl'].strip()
        append_to = 'opportunities'
    if not url:
        self.hierarchy.append('')
        return
    if url not in self.found_urls and append_to:
        obj = common.SuperDict({'url': url, 'title': title})
        self.found_urls[url] = (append_to, obj)
        self.harvest[append_to].append(obj)
    else:
        obj = self.found_urls[url][1]
    obj.setdefault('categories', [])
    if 'category' in attrs.keys():
        for i in attrs['category'].split(','):
            tmp = [j.strip() for j in i.split('/') if j.strip()]
            if tmp and tmp not in obj['categories']:
                obj['categories'].append(tmp)
    if self.hierarchy and self.hierarchy not in obj['categories']:
        obj['categories'].append(copy.copy(self.hierarchy))
    obj['tags'] = [i[0] for i in obj['categories'] if len(i) == 1]
    self.hierarchy.append('')