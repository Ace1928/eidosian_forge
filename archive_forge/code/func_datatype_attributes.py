def datatype_attributes(root, datatype):

    def _iterchildren(node, pathsofar):
        elements = []
        for child in node.iterchildren():
            if isinstance(child.tag, str) and child.tag.split('}')[1] == 'element':
                elements.append('%s/%s' % (pathsofar, child.get('name')))
                elements.extend(_iterchildren(child, '%s/%s' % (pathsofar, child.get('name'))))
            elif isinstance(child.tag, str) and child.tag.split('}')[1] == 'attribute':
                elements.append('%s/%s' % (pathsofar, child.get('name')))
            elif isinstance(child.tag, str) and child.tag.split('}')[1] == 'extension':
                ct_xpath = "/xs:schema/xs:complexType[@name='%s']" % child.get('base').split(':')[1]
                rt = node.getroottree()
                for complex_type in rt.xpath(ct_xpath, namespaces=child.nsmap):
                    same = False
                    for ancestor in child.iterancestors():
                        if ancestor.get('name') == child.get('base').split(':')[1]:
                            same = True
                            break
                    if not same:
                        elements.extend(_iterchildren(complex_type, pathsofar))
                    elements.extend(_iterchildren(child, pathsofar))
            else:
                elements.extend(_iterchildren(child, pathsofar))
        return elements
    ct_xpath = "/xs:schema/xs:complexType[@name='%s']" % datatype.split(':')[1]
    attributes = []
    for complex_type in root.xpath(ct_xpath, namespaces=root.nsmap):
        for child in complex_type.iterchildren():
            attributes.extend(_iterchildren(child, datatype))
    return attributes