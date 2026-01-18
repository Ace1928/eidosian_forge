import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
def test_debug_info(self):
    try:
        assert os.path.exists(self.debug_dest)
        t = DebugWriter.etree.parse(self.debug_dest)
        L = list(t.find('/Module/Globals'))
        assert L
        xml_globals = dict(((e.attrib['name'], e.attrib['type']) for e in L))
        self.assertEqual(len(L), len(xml_globals))
        L = list(t.find('/Module/Functions'))
        assert L
        xml_funcs = dict(((e.attrib['qualified_name'], e) for e in L))
        self.assertEqual(len(L), len(xml_funcs))
        self.assertEqual('CObject', xml_globals.get('c_var'))
        self.assertEqual('PythonObject', xml_globals.get('python_var'))
        funcnames = ('codefile.spam', 'codefile.ham', 'codefile.eggs', 'codefile.closure', 'codefile.inner')
        required_xml_attrs = ('name', 'cname', 'qualified_name')
        assert all((f in xml_funcs for f in funcnames))
        spam, ham, eggs = [xml_funcs[funcname] for funcname in funcnames]
        self.assertEqual(spam.attrib['name'], 'spam')
        self.assertNotEqual('spam', spam.attrib['cname'])
        assert self.elem_hasattrs(spam, required_xml_attrs)
        spam_locals = list(spam.find('Locals'))
        assert spam_locals
        spam_locals.sort(key=lambda e: e.attrib['name'])
        names = [e.attrib['name'] for e in spam_locals]
        self.assertEqual(list('abcd'), names)
        assert self.elem_hasattrs(spam_locals[0], required_xml_attrs)
        spam_arguments = list(spam.find('Arguments'))
        assert spam_arguments
        self.assertEqual(1, len(list(spam_arguments)))
        step_into = spam.find('StepIntoFunctions')
        spam_stepinto = [x.attrib['name'] for x in step_into]
        assert spam_stepinto
        self.assertEqual(2, len(spam_stepinto))
        assert 'puts' in spam_stepinto
        assert 'some_c_function' in spam_stepinto
    except:
        f = open(self.debug_dest)
        try:
            print(f.read())
        finally:
            f.close()
        raise