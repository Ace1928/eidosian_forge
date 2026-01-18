from typing import Dict, List
def intensify(color: str, dark_bg: bool, amount: int=64) -> str:
    if not dark_bg:
        amount = -amount
    rgb = tuple((max(0, min(255, amount + int(color[i:i + 2], 16))) for i in (1, 3, 5)))
    return '#%.2x%.2x%.2x' % rgb