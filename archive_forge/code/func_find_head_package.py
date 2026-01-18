import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def find_head_package(self, parent, name):
    self.msgin(4, 'find_head_package', parent, name)
    if '.' in name:
        i = name.find('.')
        head = name[:i]
        tail = name[i + 1:]
    else:
        head = name
        tail = ''
    if parent:
        qname = '%s.%s' % (parent.__name__, head)
    else:
        qname = head
    q = self.import_module(head, qname, parent)
    if q:
        self.msgout(4, 'find_head_package ->', (q, tail))
        return (q, tail)
    if parent:
        qname = head
        parent = None
        q = self.import_module(head, qname, parent)
        if q:
            self.msgout(4, 'find_head_package ->', (q, tail))
            return (q, tail)
    self.msgout(4, 'raise ImportError: No module named', qname)
    raise ImportError('No module named ' + qname)