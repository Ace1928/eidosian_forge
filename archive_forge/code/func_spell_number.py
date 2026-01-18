import re
def spell_number(self, num):
    if num == 0:
        return 'zero'
    parts = []
    for i in range(0, len(self.thousands)):
        if num % 1000 != 0:
            part = ''
            hundreds = num % 1000 // 100
            tens_units = num % 100
            if hundreds > 0:
                part += self.ones[hundreds] + ' hundred'
                if tens_units > 0:
                    part += ' and '
            if tens_units > 10 and tens_units < 20:
                part += self.teens[tens_units - 10]
            else:
                tens_digit = self.tens[tens_units // 10]
                ones_digit = self.ones[tens_units % 10]
                if tens_digit:
                    part += tens_digit
                if ones_digit:
                    if tens_digit:
                        part += ' '
                    part += ones_digit
            parts.append(part)
        num //= 1000
    return ' '.join(reversed(parts))