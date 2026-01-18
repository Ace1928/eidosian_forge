from __future__ import unicode_literals
def nofollow(attrs, new=False):
    href_key = (None, u'href')
    if href_key not in attrs or attrs[href_key].startswith(u'mailto:'):
        return attrs
    rel_key = (None, u'rel')
    rel_values = [val for val in attrs.get(rel_key, u'').split(u' ') if val]
    if u'nofollow' not in [rel_val.lower() for rel_val in rel_values]:
        rel_values.append(u'nofollow')
    attrs[rel_key] = u' '.join(rel_values)
    return attrs