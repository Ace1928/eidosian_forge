def is_name_continue(char: str) -> bool:
    """Check whether char is allowed in the continuation of a GraphQL name

        For internal use by the lexer only.
        """
    return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9' or (char == '_')