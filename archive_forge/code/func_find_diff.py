import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import (
import os
from glob import glob
import logging
from prov.tests import examples
import prov.model as pm
import rdflib as rl
from rdflib.compare import graph_diff
from io import BytesIO, StringIO
def find_diff(g_rdf, g0_rdf):
    graphs_equal = True
    in_both, in_first, in_second = graph_diff(g_rdf, g0_rdf)
    g1 = sorted(in_first.serialize(format='nt').splitlines())[1:]
    g2 = sorted(in_second.serialize(format='nt').splitlines())[1:]
    if len(g1) != len(g2):
        graphs_equal = False
    matching_indices = [[], []]
    for idx in range(len(g1)):
        g1_stmt = list(rl.ConjunctiveGraph().parse(BytesIO(g1[idx]), format='nt'))[0]
        match_found = False
        for idx2 in range(len(g2)):
            if idx2 in matching_indices[1]:
                continue
            g2_stmt = list(rl.ConjunctiveGraph().parse(BytesIO(g2[idx2]), format='nt'))[0]
            try:
                all_match = all([g1_stmt[i].eq(g2_stmt[i]) for i in range(3)])
            except TypeError:
                all_match = False
            if all_match:
                matching_indices[0].append(idx)
                matching_indices[1].append(idx2)
                match_found = True
                break
        if not match_found:
            graphs_equal = False
    in_first2 = rl.ConjunctiveGraph()
    for idx in range(len(g1)):
        if idx in matching_indices[0]:
            in_both.parse(BytesIO(g1[idx]), format='nt')
        else:
            in_first2.parse(BytesIO(g1[idx]), format='nt')
    in_second2 = rl.ConjunctiveGraph()
    for idx in range(len(g2)):
        if idx not in matching_indices[1]:
            in_second2.parse(BytesIO(g2[idx]), format='nt')
    return (graphs_equal, in_both, in_first2, in_second2)