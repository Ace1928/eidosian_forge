import unittest
def test_invalid_childname_before(self):
    from kivy.lang import Builder, ParserException
    try:
        Builder.load_string('\nWidget:\n    thecursor.Cursor:\n    FloatLayout:\n        size: self.parent.size\n        Button:\n            text: "text"\n            size_hint:(0.1, 0.1)\n            pos_hint:{\'x\':0.45, \'y\':0.45}\n            ')
        self.fail('Invalid children name')
    except ParserException:
        pass