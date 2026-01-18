import spherogram
import string
def from_gauss_code(code):
    """
    This is the basic unsigned/unoriented variant.

    >>> L = from_gauss_code(unknot_28)
    >>> L.simplify('pickup')
    True
    >>> L
    <Link: 0 comp; 0 cross>
    >>> L.unlinked_unknot_components
    1
    """
    n = len(code)
    labels = list(range(1, n + 1))
    code_to_label = dict(zip(code, labels))
    label_to_code = dict(zip(labels, code))
    dt = []
    for i in range(1, n, 2):
        a = label_to_code[i]
        j = code_to_label[-a]
        if a < 0:
            j = -j
        dt.append(j)
    return spherogram.Link(f'DT: {dt}')