from __future__ import unicode_literals
def target_blank(attrs, new=False):
    href_key = (None, u'href')
    if attrs[href_key].startswith(u'mailto:'):
        return attrs
    attrs[None, u'target'] = u'_blank'
    return attrs