from collections import defaultdict
def final_and(a):
    """
        Adds remaining points that are not in the intersection, not in the
        neighboring alignments but in the original *e2f* and *f2e* alignments
        """
    for e_new in range(srclen):
        for f_new in range(trglen):
            if e_new not in aligned and f_new not in aligned and ((e_new, f_new) in union):
                alignment.add((e_new, f_new))
                aligned['e'].add(e_new)
                aligned['f'].add(f_new)