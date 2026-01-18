import os
import re
import shelve
import sys
import nltk.data
def process_bundle(rels):
    """
    Given a list of relation metadata bundles, make a corresponding
    dictionary of concepts, indexed by the relation name.

    :param rels: bundle of metadata needed for constructing a concept
    :type rels: list(dict)
    :return: a dictionary of concepts, indexed by the relation name.
    :rtype: dict(str): Concept
    """
    concepts = {}
    for rel in rels:
        rel_name = rel['rel_name']
        closures = rel['closures']
        schema = rel['schema']
        filename = rel['filename']
        concept_list = clause2concepts(filename, rel_name, schema, closures)
        for c in concept_list:
            label = c.prefLabel
            if label in concepts:
                for data in c.extension:
                    concepts[label].augment(data)
                concepts[label].close()
            else:
                concepts[label] = c
    return concepts