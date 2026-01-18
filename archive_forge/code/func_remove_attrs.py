from ..host        import HostLanguage
def remove_attrs(node):
    from ..termorcurie import termname
    if options.host_language in [HostLanguage.html5, HostLanguage.xhtml5, HostLanguage.xhtml]:
        if node.tagName != 'meta' and node.hasAttribute('content'):
            generate_warning(node, 'content')
        if node.tagName != 'link' and node.hasAttribute('rel'):
            if node.tagName == 'a':
                vals = node.getAttribute('rel').strip().split()
                if len(vals) != 0:
                    final_vals = [v for v in vals if not termname.match(v)]
                    if len(final_vals) != 0:
                        generate_warning(node, 'rel')
            else:
                generate_warning(node, 'rel')
        for attr in non_lite_attributes_html:
            if node.hasAttribute(attr):
                generate_warning(node, attr)
    else:
        for attr in non_lite_attributes:
            if node.hasAttribute(attr):
                generate_warning(node, attr)