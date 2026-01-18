def set_about(node):
    if has_one_of_attributes(node, 'rel', 'rev'):
        if not has_one_of_attributes(top, 'about', 'src'):
            node.setAttribute('about', '')
    elif not has_one_of_attributes(node, 'href', 'resource', 'about', 'src'):
        node.setAttribute('about', '')