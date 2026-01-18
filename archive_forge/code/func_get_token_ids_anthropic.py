from typing import Any, List
def get_token_ids_anthropic(text: str) -> List[int]:
    """Get the token ids for a string of text."""
    client = _get_anthropic_client()
    tokenizer = client.get_tokenizer()
    encoded_text = tokenizer.encode(text)
    return encoded_text.ids