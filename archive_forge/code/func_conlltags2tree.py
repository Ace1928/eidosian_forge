import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def conlltags2tree(sentence, chunk_types=('NP', 'PP', 'VP'), root_label='S', strict=False):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for word, postag, chunktag in sentence:
        if chunktag is None:
            if strict:
                raise ValueError('Bad conll tag sequence')
            else:
                tree.append((word, postag))
        elif chunktag.startswith('B-'):
            tree.append(Tree(chunktag[2:], [(word, postag)]))
        elif chunktag.startswith('I-'):
            if len(tree) == 0 or not isinstance(tree[-1], Tree) or tree[-1].label() != chunktag[2:]:
                if strict:
                    raise ValueError('Bad conll tag sequence')
                else:
                    tree.append(Tree(chunktag[2:], [(word, postag)]))
            else:
                tree[-1].append((word, postag))
        elif chunktag == 'O':
            tree.append((word, postag))
        else:
            raise ValueError(f'Bad conll tag {chunktag!r}')
    return tree