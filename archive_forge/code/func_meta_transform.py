def meta_transform(html, options, _state):
    """
    @param html: a DOM node for the top level html element
    @param options: invocation options
    @type options: L{Options<pyRdfa.options>}
    @param state: top level execution state
    @type state: L{State<pyRdfa.state>}
    """
    from ..host import HostLanguage
    if not options.host_language in [HostLanguage.xhtml, HostLanguage.html5, HostLanguage.xhtml5]:
        return
    for meta in html.getElementsByTagName('meta'):
        if meta.hasAttribute('name') and (not meta.hasAttribute('property')):
            meta.setAttribute('property', meta.getAttribute('name'))