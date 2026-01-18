def createCodeIndenter(output):
    global _indenter
    _indenter = _IndentedCodeWriter(output)