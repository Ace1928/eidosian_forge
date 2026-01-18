import logging
def codeblock(self, code):
    """
        Literal code blocks are introduced by ending a paragraph with
        the special marker ::.  The literal block must be indented
        (and, like all paragraphs, separated from the surrounding
        ones by blank lines).
        """
    self.start_codeblock()
    self.doc.writeln(code)
    self.end_codeblock()