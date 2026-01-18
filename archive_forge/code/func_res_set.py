def res_set(node):
    return True in [node.hasAttribute(a) for a in ['resource', 'about', 'href', 'src']]