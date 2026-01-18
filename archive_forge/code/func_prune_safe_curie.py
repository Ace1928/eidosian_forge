def prune_safe_curie(node, name):
    if node.hasAttribute(name):
        av = node.getAttribute(name)
        if av == '[]':
            node.removeAttribute(name)
            node.setAttribute(name + '_pruned', '')
            msg = 'Attribute @%s uses an empty safe CURIE; the attribute is ignored' % name
            options.add_warning(msg, node=node)