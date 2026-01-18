import spherogram
def all_cross_strands(tangle):
    """
    Return all the strands but without duplicate in the opposite direction,
    starting at position 0 and going clockwise, and then components that
    don't intersect the boundary.
    """
    other_ends_seen = []
    strands = []
    strands_with_ends = []
    loops = []
    tm, tn = tangle.boundary
    clockwise_order = list(range(tm))
    clockwise_order.extend(reversed(range(tm, tm + tn)))
    for i in clockwise_order:
        if i not in other_ends_seen:
            strand = cross_strand(tangle, i)
            cs = strand[-1]
            end = tangle.adjacent.index((cs[0], (cs[1] + 2) % 4))
            if end not in other_ends_seen:
                strands.append(strand)
                strands_with_ends.append((strand, end))
                other_ends_seen.append(end)
    orientations, over_or_under = crossing_orientations(strands)
    cs_seen = [cs for strand in strands for cs in strand]
    seen_once = set((cs[0] for cs in cs_seen))
    for crossing in orientations:
        seen_once.remove(crossing)
    for strand in strands:
        for cs in strand:
            if cs[0] in seen_once:
                loop = loop_strand((cs[0], (cs[1] + 1) % 4))
                loops.append(loop)
                cs_seen.extend(loop)
                for loop_cs in loop:
                    if loop_cs[0] in seen_once:
                        for seen_cs in cs_seen:
                            if loop_cs[0] == seen_cs[0]:
                                orientation = (loop_cs[1] - seen_cs[1]) % 4
                                if orientation == 3:
                                    orientation = -1
                                orientations[loop_cs[0]] = orientation
                                over_or_under[loop_cs[0]] = loop_cs[1] % 2
                                seen_once.remove(loop_cs[0])
                                break
                    else:
                        seen_once.add(loop_cs[0])
    while len(orientations) < len(tangle.crossings):
        for loop in loops:
            for cs in loop:
                if cs[0] in seen_once:
                    loop = loop_strand((cs[0], (cs[1] + 1) % 4))
                    loops.append(loop)
                    cs_seen.extend(loop)
                    for loop_cs in loop:
                        if loop_cs[0] in seen_once:
                            for seen_cs in cs_seen:
                                if loop_cs[0] == seen_cs[0]:
                                    orientation = (loop_cs[1] - seen_cs[1]) % 4
                                    if orientation == 3:
                                        orientation = -1
                                    orientations[loop_cs[0]] = orientation
                                    over_or_under[loop_cs[0]] = loop_cs[1] % 2
                                    seen_once.remove(loop_cs[0])
                                    break
                        else:
                            seen_once.add(loop_cs[0])
    crossings = [x[0] for x in cs_seen]
    crossing_order = []
    for c in crossings:
        if c not in crossing_order:
            crossing_order.append(c)
    return (strands_with_ends, loops, orientations, crossing_order, over_or_under)