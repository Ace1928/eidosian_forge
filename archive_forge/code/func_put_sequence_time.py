from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def put_sequence_time(G, att_name):
    current_date = date(2015, 1, 1)
    for e in G.edges(data=True):
        current_date += timedelta(days=1)
        e[2][att_name] = current_date
    return G