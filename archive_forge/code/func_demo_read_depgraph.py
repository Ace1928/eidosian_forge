from itertools import chain
from nltk.internals import Counter
def demo_read_depgraph():
    from nltk.parse.dependencygraph import DependencyGraph
    dg1 = DependencyGraph('Esso       NNP     2       SUB\nsaid       VBD     0       ROOT\nthe        DT      5       NMOD\nWhiting    NNP     5       NMOD\nfield      NN      6       SUB\nstarted    VBD     2       VMOD\nproduction NN      6       OBJ\nTuesday    NNP     6       VMOD\n')
    dg2 = DependencyGraph('John    NNP     2       SUB\nsees    VBP     0       ROOT\nMary    NNP     2       OBJ\n')
    dg3 = DependencyGraph('a       DT      2       SPEC\nman     NN      3       SUBJ\nwalks   VB      0       ROOT\n')
    dg4 = DependencyGraph('every   DT      2       SPEC\ngirl    NN      3       SUBJ\nchases  VB      0       ROOT\na       DT      5       SPEC\ndog     NN      3       OBJ\n')
    depgraphs = [dg1, dg2, dg3, dg4]
    for dg in depgraphs:
        print(FStructure.read_depgraph(dg))