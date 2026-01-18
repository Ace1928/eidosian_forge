from io import StringIO
from re import finditer, search
from textwrap import dedent
import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
def test_graphviz_toy():
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini', random_state=2)
    clf.fit(X, y)
    contents1 = export_graphviz(clf, out_file=None)
    contents2 = 'digraph Tree {\nnode [shape=box, fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\nvalue = [3, 3]"] ;\n1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n}'
    assert contents1 == contents2
    contents1 = export_graphviz(clf, filled=True, impurity=False, proportion=True, special_characters=True, rounded=True, out_file=None, fontname='sans')
    contents2 = 'digraph Tree {\nnode [shape=box, style="filled, rounded", color="black", fontname="sans"] ;\nedge [fontname="sans"] ;\n0 [label=<x<SUB>0</SUB> &le; 0.0<br/>samples = 100.0%<br/>value = [0.5, 0.5]>, fillcolor="#ffffff"] ;\n1 [label=<samples = 50.0%<br/>value = [1.0, 0.0]>, fillcolor="#e58139"] ;\n0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n2 [label=<samples = 50.0%<br/>value = [0.0, 1.0]>, fillcolor="#399de5"] ;\n0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n}'
    assert contents1 == contents2
    contents1 = export_graphviz(clf, max_depth=0, class_names=True, out_file=None)
    contents2 = 'digraph Tree {\nnode [shape=box, fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\nvalue = [3, 3]\\nclass = y[0]"] ;\n1 [label="(...)"] ;\n0 -> 1 ;\n2 [label="(...)"] ;\n0 -> 2 ;\n}'
    assert contents1 == contents2
    contents1 = export_graphviz(clf, max_depth=0, filled=True, out_file=None, node_ids=True)
    contents2 = 'digraph Tree {\nnode [shape=box, style="filled", color="black", fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="node #0\\nx[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\nvalue = [3, 3]", fillcolor="#ffffff"] ;\n1 [label="(...)", fillcolor="#C0C0C0"] ;\n0 -> 1 ;\n2 [label="(...)", fillcolor="#C0C0C0"] ;\n0 -> 2 ;\n}'
    assert contents1 == contents2
    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=2, criterion='gini', random_state=2)
    clf = clf.fit(X, y2, sample_weight=w)
    contents1 = export_graphviz(clf, filled=True, impurity=False, out_file=None)
    contents2 = 'digraph Tree {\nnode [shape=box, style="filled", color="black", fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="x[0] <= 0.0\\nsamples = 6\\nvalue = [[3.0, 1.5, 0.0]\\n[3.0, 1.0, 0.5]]", fillcolor="#ffffff"] ;\n1 [label="samples = 3\\nvalue = [[3, 0, 0]\\n[3, 0, 0]]", fillcolor="#e58139"] ;\n0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n2 [label="x[0] <= 1.5\\nsamples = 3\\nvalue = [[0.0, 1.5, 0.0]\\n[0.0, 1.0, 0.5]]", fillcolor="#f1bd97"] ;\n0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n3 [label="samples = 2\\nvalue = [[0, 1, 0]\\n[0, 1, 0]]", fillcolor="#e58139"] ;\n2 -> 3 ;\n4 [label="samples = 1\\nvalue = [[0.0, 0.5, 0.0]\\n[0.0, 0.0, 0.5]]", fillcolor="#e58139"] ;\n2 -> 4 ;\n}'
    assert contents1 == contents2
    clf = DecisionTreeRegressor(max_depth=3, min_samples_split=2, criterion='squared_error', random_state=2)
    clf.fit(X, y)
    contents1 = export_graphviz(clf, filled=True, leaves_parallel=True, out_file=None, rotate=True, rounded=True, fontname='sans')
    contents2 = 'digraph Tree {\nnode [shape=box, style="filled, rounded", color="black", fontname="sans"] ;\ngraph [ranksep=equally, splines=polyline] ;\nedge [fontname="sans"] ;\nrankdir=LR ;\n0 [label="x[0] <= 0.0\\nsquared_error = 1.0\\nsamples = 6\\nvalue = 0.0", fillcolor="#f2c09c"] ;\n1 [label="squared_error = 0.0\\nsamples = 3\\nvalue = -1.0", fillcolor="#ffffff"] ;\n0 -> 1 [labeldistance=2.5, labelangle=-45, headlabel="True"] ;\n2 [label="squared_error = 0.0\\nsamples = 3\\nvalue = 1.0", fillcolor="#e58139"] ;\n0 -> 2 [labeldistance=2.5, labelangle=45, headlabel="False"] ;\n{rank=same ; 0} ;\n{rank=same ; 1; 2} ;\n}'
    assert contents1 == contents2
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y_degraded)
    contents1 = export_graphviz(clf, filled=True, out_file=None)
    contents2 = 'digraph Tree {\nnode [shape=box, style="filled", color="black", fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="gini = 0.0\\nsamples = 6\\nvalue = 6.0", fillcolor="#ffffff"] ;\n}'