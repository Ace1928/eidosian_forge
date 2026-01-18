@classmethod
def from_ReactionSystem(cls, rsys, color_categories=True):
    idx_rxn_pairs, unconsidered_ri = getattr(rsys, cls._rsys_meth)()
    colors = rsys._category_colors() if color_categories else {}
    missing = [not rsys.substance_participation(sk) for sk in rsys.substances]
    return (cls(idx_rxn_pairs, rsys.substances, colors=colors, missing=missing), unconsidered_ri)