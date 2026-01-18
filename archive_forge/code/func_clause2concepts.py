import os
import re
import shelve
import sys
import nltk.data
def clause2concepts(filename, rel_name, schema, closures=[]):
    """
    Convert a file of Prolog clauses into a list of ``Concept`` objects.

    :param filename: filename containing the relations
    :type filename: str
    :param rel_name: name of the relation
    :type rel_name: str
    :param schema: the schema used in a set of relational tuples
    :type schema: list
    :param closures: closure properties for the extension of the concept
    :type closures: list
    :return: a list of ``Concept`` objects
    :rtype: list
    """
    concepts = []
    subj = 0
    pkey = schema[0]
    fields = schema[1:]
    records = _str2records(filename, rel_name)
    if not filename in not_unary:
        concepts.append(unary_concept(pkey, subj, records))
    for field in fields:
        obj = schema.index(field)
        concepts.append(binary_concept(field, closures, subj, obj, records))
    return concepts