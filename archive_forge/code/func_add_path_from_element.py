import re
def add_path_from_element(self, el):
    tag = _strip_xml_ns(el.tag)
    parse_fn = getattr(self, '_parse_%s' % tag.lower(), None)
    if not callable(parse_fn):
        return False
    parse_fn(el)
    if 'transform' in el.attrib:
        self.transforms[-1] = _transform(el.attrib['transform'])
    return True