import collections
import itertools
import sys
import rdflib.extras.cmdlineutils
from rdflib import RDF, RDFS, XSD
def rdfs2dot(g, stream, opts={}):
    """
    Convert the RDFS schema in a graph
    writes the dot output to the stream
    """
    fields = collections.defaultdict(set)
    nodes = {}

    def node(nd):
        if nd not in nodes:
            nodes[nd] = 'node%d' % len(nodes)
        return nodes[nd]

    def label(xx, grf):
        lbl = grf.value(xx, RDFS.label)
        if lbl is None:
            try:
                lbl = grf.namespace_manager.compute_qname(xx)[2]
            except Exception:
                pass
        return lbl
    stream.write('digraph { \n node [ fontname="DejaVu Sans" ] ; \n')
    for x in g.subjects(RDF.type, RDFS.Class):
        n = node(x)
    for x, y in g.subject_objects(RDFS.subClassOf):
        x = node(x)
        y = node(y)
        stream.write('\t%s -> %s [ color=%s ] ;\n' % (y, x, ISACOLOR))
    for x in g.subjects(RDF.type, RDF.Property):
        for a, b in itertools.product(g.objects(x, RDFS.domain), g.objects(x, RDFS.range)):
            if b in XSDTERMS or b == RDFS.Literal:
                l_ = label(b, g)
                if b == RDFS.Literal:
                    l_ = 'literal'
                fields[node(a)].add((label(x, g), l_))
            else:
                stream.write('\t%s -> %s [ color=%s, label="%s" ];\n' % (node(a), node(b), EDGECOLOR, label(x, g)))
    for u, n in nodes.items():
        stream.write('# %s %s\n' % (u, n))
        f = ["<tr><td align='left'>%s</td><td>%s</td></tr>" % x for x in sorted(fields[n])]
        opstr = "%s [ shape=none, color=%s label=< <table color='#666666'" + " cellborder='0' cellspacing='0' border='1'><tr>" + "<td colspan='2' bgcolor='grey'><B>%s</B></td>" + '</tr>%s</table> > ] \n'
        stream.write(opstr % (n, NODECOLOR, label(u, g), ''.join(f)))
    stream.write('}\n')