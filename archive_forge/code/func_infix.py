def infix(bp, func):
    """
    Create an infix operator, given a binding power and a function that
    evaluates the node.
    """

    class Operator(TokenBase):
        lbp = bp

        def led(self, left, parser):
            self.first = left
            self.second = parser.expression(bp)
            return self

        def eval(self, context):
            try:
                return func(context, self.first, self.second)
            except Exception:
                return False
    return Operator