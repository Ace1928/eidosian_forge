from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
def inst_vars(self, edge):
    return {var: logic.unique_variable() for var in edge.lhs().variables() if var.name.startswith('@')}