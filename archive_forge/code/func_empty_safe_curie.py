def empty_safe_curie(node, options, state):
    """
    Remove the attributes whose value is an empty safe curie. It also adds an 'artificial' flag, ie, an
    attribute (called 'emptysc') into the node to signal that there _is_ an attribute with an ignored
    safe curie value. The name of the attribute is 'about_pruned' or 'resource_pruned'.
    
    @param node: a DOM node for the top level element
    @param options: invocation options
    @type options: L{Options<pyRdfa.options>}
    @param state: top level execution state
    @type state: L{State<pyRdfa.state>}
    """

    def prune_safe_curie(node, name):
        if node.hasAttribute(name):
            av = node.getAttribute(name)
            if av == '[]':
                node.removeAttribute(name)
                node.setAttribute(name + '_pruned', '')
                msg = 'Attribute @%s uses an empty safe CURIE; the attribute is ignored' % name
                options.add_warning(msg, node=node)
    prune_safe_curie(node, 'about')
    prune_safe_curie(node, 'resource')
    for n in node.childNodes:
        if n.nodeType == node.ELEMENT_NODE:
            empty_safe_curie(n, options, state)