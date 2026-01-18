import os
import re
import shelve
import sys
import nltk.data
def binary_concept(label, closures, subj, obj, records):
    """
    Make a binary concept out of the primary key and another field in a record.

    A record is a list of entities in some relation, such as
    ``['france', 'paris']``, where ``'france'`` is acting as the primary
    key, and ``'paris'`` stands in the ``'capital_of'`` relation to
    ``'france'``.

    More generally, given a record such as ``['a', 'b', 'c']``, where
    label is bound to ``'B'``, and ``obj`` bound to 1, the derived
    binary concept will have label ``'B_of'``, and its extension will
    be a set of pairs such as ``('a', 'b')``.


    :param label: the base part of the preferred label for the concept
    :type label: str
    :param closures: closure properties for the extension of the concept
    :type closures: list
    :param subj: position in the record of the subject of the predicate
    :type subj: int
    :param obj: position in the record of the object of the predicate
    :type obj: int
    :param records: a list of records
    :type records: list of lists
    :return: ``Concept`` of arity 2
    :rtype: Concept
    """
    if not label == 'border' and (not label == 'contain'):
        label = label + '_of'
    c = Concept(label, arity=2, closures=closures, extension=set())
    for record in records:
        c.augment((record[subj], record[obj]))
    c.close()
    return c