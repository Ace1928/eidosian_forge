import unittest
def test_fork_binding(self):
    from kivy.lang import Builder
    from kivy.uix.behaviors.knspace import knspace
    w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\n<MyComplexWidget@NamedLabel>:\n    knsname: 'root_label'\n    text: 'Hello'\n    NamedLabel:\n        id: child\n        knsname: 'child_label'\n        text: self.knspace.root_label.text if self.knspace.root_label else ''\n\nBoxLayout:\n    MyComplexWidget:\n        knspace: 'fork'\n        id: first\n    MyComplexWidget:\n        knspace: 'fork'\n        id: second\n")
    self.assertEqual(w.ids.first.ids.child.text, 'Hello')
    self.assertEqual(w.ids.second.ids.child.text, 'Hello')
    self.assertEqual(w.ids.first.knspace.child_label.text, 'Hello')
    self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
    w.ids.first.text = 'Goodbye'
    self.assertEqual(w.ids.first.ids.child.text, 'Goodbye')
    self.assertEqual(w.ids.second.ids.child.text, 'Hello')
    self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye')
    self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
    first = w.ids.first.knspace
    w.ids.first.knspace = w.ids.second.knspace
    w.ids.second.knspace = first
    self.assertEqual(w.ids.first.ids.child.text, 'Goodbye')
    self.assertEqual(w.ids.second.ids.child.text, 'Hello')
    self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye')
    self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
    w.ids.first.text = 'Goodbye2'
    self.assertEqual(w.ids.first.ids.child.text, 'Goodbye2')
    self.assertEqual(w.ids.second.ids.child.text, 'Hello')
    self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye2')
    self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
    w.ids.first.knspace.root_label.text = 'Goodbye3'
    self.assertEqual(w.ids.first.ids.child.text, 'Goodbye3')
    self.assertEqual(w.ids.second.ids.child.text, 'Hello')
    self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye3')
    self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')