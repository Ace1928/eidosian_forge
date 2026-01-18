import unittest
import commonmark
def render_rst(self, test_str):
    ast = self.parser.parse(test_str)
    rst = self.renderer.render(ast)
    return rst