import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def tangle_neighborhood(link, crossing, radius, return_gluings=True, hull=False):
    """
    Splits a link into two tangles along a ball around a crossing of the given
    radius.  Destroys original link.  This might not generate an actual tangle;
    the graph metric ball will usually have multiple boundary components.
    """
    crossings, adjacent = crossing_ball(crossing, radius)
    crossings = list(crossings)
    opposites = list(reversed(map(lambda x: x.opposite(), adjacent)))
    outside_crossings = [c for c in link.crossings if c not in crossings]
    if len(outside_crossings) == 0:
        raise Exception('Neighborhood is entire link')
    n = len(adjacent) / 2
    if hull:
        comps = list(boundary_components(link, crossing, radius))
        largest_comp = max(comps)
        sides = dict([(cslabel(cross_strand), cross_strand) for cross_strand in adjacent])
        c = largest_comp.pop()
        cs = choice(c.crossing_strands())
        exit_strand = meander(cs, sides)[1]
        exit_strand = exit_strand[0].crossing_strands()[exit_strand[1]]
        main_boundary_comp = trace_boundary_component(exit_strand, adjacent)
        print('main_boundary_comp' + str(main_boundary_comp))
        print('all comps: ' + str(comps))
        comps.remove(largest_comp)
        for comp in comps:
            print('crossings: ' + str(crossings))
            print('filling in comp:' + str(comp))
            print('adjacent: ' + str(adjacent))
            c = comp.pop()
            cs = choice(c.crossing_strands())
            print('cs: ' + str(cs))
            exit_strand = meander(cs, sides)[1]
            exit_strand = exit_strand[0].crossing_strands()[exit_strand[1]]
            print('exit_strand: ' + str(exit_strand))
            bound_comp = trace_boundary_component(exit_strand, adjacent)
            print('traced component: ' + str(bound_comp))
            if exit_strand not in main_boundary_comp:
                for x in bound_comp:
                    adjacent.remove(x)
                print('updated adjacent: ' + str(adjacent))
                crossings.append(c)
                crossings.extend(list(comp))
    adjacent[n:] = reversed(adjacent[n:])
    opposites[n:] = reversed(opposites[n:])
    gluings = []
    seen_cs = []
    for cs in adjacent:
        if cs in seen_cs:
            continue
        next_cross = cs.next_corner()
        while next_cross not in adjacent:
            next_cross = next_cross.next_corner()
        if next_cross != cs:
            gluings.append((adjacent.index(cs), adjacent.index(next_cross)))
            seen_cs.append(next_cross)
    clear_orientations(crossings)
    clear_orientations(outside_crossings)
    gluings.sort()
    if return_gluings:
        return (Tangle(n, crossings, adjacent), Tangle(n, outside_crossings, opposites), gluings)
    else:
        return (Tangle(n, crossings, adjacent), Tangle(n, outside_crossings, opposites))