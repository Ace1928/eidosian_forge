from io import StringIO
from .host import accept_embedded_rdf_xml, accept_embedded_turtle
from .utils import return_XML
import sys
def handle_embeddedRDF(node, graph, state):
    """
    Handles embedded RDF. There are two possibilities:
    
     - the file is one of the XML dialects that allows for an embedded RDF/XML portion. See the L{host.accept_embedded_rdf_xml} for those (a typical example is SVG). 
     - the file is HTML and there is a turtle portion in the C{<script>} element with type text/turtle. 
    
    @param node: a DOM node for the top level element
    @param graph: target rdf graph
    @type graph: RDFLib's Graph object instance
    @param state: the inherited state (namespaces, lang, etc)
    @type state: L{state.ExecutionContext}
    @return: whether an RDF/XML or turtle content has been detected or not. If TRUE, the RDFa processing should not occur on the node and its descendents. 
    @rtype: Boolean
    """

    def _get_literal(Pnode):
        """
        Get the full text
        @param Pnode: DOM Node
        @return: string
        """
        rc = ''
        for node in Pnode.childNodes:
            if node.nodeType in [node.TEXT_NODE, node.CDATA_SECTION_NODE]:
                rc = rc + node.data
        return rc.replace('<![CDATA[', '').replace(']]>', '')
    if state.options.embedded_rdf:
        if state.options.host_language in accept_embedded_turtle and node.nodeName.lower() == 'script':
            if node.hasAttribute('type') and node.getAttribute('type') == 'text/turtle':
                content = _get_literal(node)
                rdf = StringIO(content)
                try:
                    graph.parse(rdf, format='n3', publicID=state.base)
                    state.options.add_info('The output graph includes triples coming from an embedded Turtle script')
                except:
                    _type, value, _traceback = sys.exc_info()
                    state.options.add_error('Embedded Turtle content could not be parsed (problems with %s?); ignored' % value)
            return True
        elif state.options.host_language in accept_embedded_rdf_xml and node.localName == 'RDF' and (node.namespaceURI == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'):
            rdf = StringIO(return_XML(state, node))
            try:
                graph.parse(rdf)
                state.options.add_info('The output graph includes triples coming from an embedded RDF/XML subtree')
            except:
                _type, value, _traceback = sys.exc_info()
                state.options.add_error('Embedded RDF/XML content could not parsed (problems with %s?); ignored' % value)
            return True
        else:
            return False
    else:
        return False